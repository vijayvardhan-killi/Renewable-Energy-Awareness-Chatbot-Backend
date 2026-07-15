import { useState } from "react";

const useRouter = () => {
  const [currentPage, setCurrentPage] = useState('home');

  const navigate = (page) => {
    setCurrentPage(page);
  };

  return { currentPage, navigate };
};

export default useRouter;
